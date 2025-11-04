import time
import pytest
from pageObjects.homePage import HomePage
from pageObjects.SignUp_LoginPage import SignUp_LoginPage
from testCases.BaseTest import BaseTest
from utilities.readProperties import ReadConfig
from utilities import XUtils
import os
import random
import string
import platform


@pytest.mark.usefixtures("load_common_info")
class Test_registerUser_excel_ddt(BaseTest):
    # baseUrl = ReadConfig.getApplicationUrl()
    os_info = platform.system()
    if os_info == "Windows":
        path = f"{os.getcwd()}\\TestData\\Book2.xlsx"
    elif os_info == "Linux":
        path = f"{os.getcwd()}/TestData/Book3.xlsx"
    elif os_info == "Darwin":  # macOS is identified as 'Darwin'
        path = f"{os.getcwd()}/TestData/Book4.xlsx"
    else:
        raise Exception(f"Unsupported OS: {os_info}")


    @pytest.fixture(autouse=True)
    def class_setup(self, load_common_info):
        self.baseUrl = load_common_info["baseurl"]
        self.email = load_common_info["email"]
        self.password = load_common_info["password"]

    #this generate random string
    def random_string(self, length):
        """
        Generate a random string of specified length.

        Args:
        length (int): The length of the random string to generate.

        Returns:
        str: A random string of the specified length.
        """
        letters = string.ascii_letters
        return ''.join(random.choice(letters) for _ in range(length))

    @pytest.mark.regression
    def test_login_user_ex(self, setup):
        self.logger.info("*********************Started**************************")
        self.logger.info("*********************test_login_user**************************")
        self.driver = setup
        self.hp = HomePage(self.driver)
        self.sp = SignUp_LoginPage(self.driver)

        print("#############################################")
        print(XUtils.getRowCount(self.path, "Sheet1"))
        self.rows = XUtils.getRowCount(self.path, "Sheet1")

        # Track failed login attempts to assert at the end
        failed_logins = []

        for r in range(2, self.rows + 1):
            self.hp.navigateToPage(self.baseUrl)
            time.sleep(2)
            self.hp.clickOnLink()
            time.sleep(3)
            self.email = XUtils.readData(self.path, 'Sheet1', r, 1)
            self.password = int(XUtils.readData(self.path, 'Sheet1', r, 2))
            self.sp.enterEmail(self.email)
            self.sp.enterPassword(self.password)
            self.sp.clickOnLoginButton()
            time.sleep(4)
            expected_url = "https://automationexercise.com/"
            actual_url = self.hp.getUrl()

            if expected_url != actual_url:
                self.driver.save_screenshot(f'Screenshot/s_{time.time()}.png')
                # Record the failure but don't fail the test yet
                failed_logins.append(f"{self.email} with password {self.password}")
                self.logger.warning(f"Login failed for {self.email}")
                # Don't try to logout if login failed
                continue

            # Only try to logout if login succeeded
            self.logger.info(f"Login successful for {self.email}")
            try:
                self.hp.clickOnLink(link='logout')
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Failed to logout for {self.email}: {str(e)}")
                self.driver.save_screenshot(f'Screenshot/logout_failed_{time.time()}.png')

        # After testing all credentials, report failures if any
        if failed_logins:
            assert False, f"The following logins failed: {', '.join(failed_logins)}"

        self.logger.info("*********************End**************************")
        self.logger.info("*********************TestEnd**************************")

    # @pytest.mark.regression
    # def test_login_user_ex(self, setup):
    #     self.logger.info("*********************Started**************************")
    #     self.logger.info("*********************test_login_user**************************")
    #     self.driver = setup
    #     self.hp = HomePage(self.driver)
    #     self.sp = SignUp_LoginPage(self.driver)
    #     print("#############################################")
    #     print(XUtils.getRowCount(self.path, "Sheet1"))
    #     self.rows = XUtils.getRowCount(self.path, "Sheet1")
    #
    #     for r in range(2, self.rows + 1):
    #         self.hp.navigateToPage(self.baseUrl)
    #         time.sleep(2)
    #         self.hp.clickOnLink()
    #         time.sleep(3)
    #         self.email = XUtils.readData(self.path, 'Sheet1', r, 1)
    #         self.password = int(XUtils.readData(self.path, 'Sheet1', r, 2))
    #         self.sp.enterEmail(self.email)
    #         self.sp.enterPassword(self.password)
    #         self.sp.clickOnLoginButton()
    #         time.sleep(4)
    #         expected_url = "https://automationexercise.com/"
    #         actual_url = self.hp.getUrl()
    #
    #         # time.sleep(500)
    #         if expected_url != actual_url:
    #             self.driver.save_screenshot(f'Screenshot/s_{time.time()}.png')
    #             assert expected_url == actual_url, f"{actual_url} for {self.email} and {self.password} login failed"
    #         time.sleep(2)
    #         self.hp.clickOnLink(link='logout')
    #     # time.sleep(50000)
    #     self.logger.info("*********************End**************************")
    #     self.logger.info("*********************TestEnd**************************")
